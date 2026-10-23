import assert from 'node:assert/strict';
import { once } from 'node:events';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { JevAPIError, JevClient } from '../dist/index.js';

const fixtures = JSON.parse(readFileSync(new URL('../../../tests/fixtures/decision-responses.json', import.meta.url))).valid;
const decision = (requestId, name = 'boolean') => ({...fixtures[name], request_id: requestId});
const request = (requestId) => ({request_id: requestId, model: 'fixture', input: {text: 'x'}});
const isInvalidResponse = (error) => error instanceof JevAPIError &&
  error.status === 502 && error.code === 'invalid_response';

async function endpoint(t, respond, prefix = '') {
  const requests = [];
  const server = createServer(async (incoming, response) => {
    const chunks = [];
    for await (const chunk of incoming) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString());
    requests.push({path: incoming.url, method: incoming.method, body});
    const reply = await respond(body, requests.length);
    response.writeHead(reply.status ?? 200, {'Content-Type': 'application/json'});
    response.end(JSON.stringify(reply.body));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  const baseURL = `http://127.0.0.1:${server.address().port}${prefix}`;
  return {requests, client: new JevClient({baseURL, timeoutMs: 2000})};
}

for (const name of ['boolean', 'partial', 'all_failed']) {
  for (const prefix of ['', '/plugins/jev-runtime']) {
    test(`rejects mismatched request ID for ${name} at ${prefix || 'gateway'}`, async (t) => {
      const {client, requests} = await endpoint(t, () => ({body: decision('another-request', name)}), prefix);
      await assert.rejects(client.decide(request('my-request')), isInvalidResponse);
      assert.deepEqual(requests, [{path: `${prefix}/v1/decisions`, method: 'POST', body: request('my-request')}]);
    });
  }

  test(`preserves matching request ID and ${name} result`, async (t) => {
    const expected = decision('caller:请求.1', name);
    const {client, requests} = await endpoint(t, (body) => ({body: decision(body.request_id, name)}));
    assert.deepEqual(await client.decide(request(expected.request_id)), expected);
    assert.equal(requests.length, 1);
  });
}

for (const mode of ['omitted', 'undefined', 'null']) {
  test(`accepts a server-generated request ID when the supplied ID is ${mode}`, async (t) => {
    const expected = decision('dec-server-generated');
    const {client, requests} = await endpoint(t, () => ({body: expected}));
    const payload = request(mode === 'null' ? null : undefined);
    if (mode === 'omitted') delete payload.request_id;
    assert.deepEqual(await client.decide(payload), expected);
    assert.deepEqual(requests[0].body, JSON.parse(JSON.stringify(payload)));
    assert.equal(requests.length, 1);
  });
}

test('captures the submitted ID before the caller mutates the in-flight object', async (t) => {
  const {client, requests} = await endpoint(t, (body) => ({body: decision(body.request_id)}));
  const payload = request('original-id');
  const pending = client.decide(payload);
  payload.request_id = 'changed-id';
  assert.equal((await pending).request_id, 'original-id');
  assert.equal(requests[0].body.request_id, 'original-id');
});

test('in-flight mutation cannot make a mismatched response appear correlated', async (t) => {
  const {client, requests} = await endpoint(t, () => ({body: decision('another-request')}));
  const payload = request('original-id');
  const pending = client.decide(payload);
  payload.request_id = 'another-request';
  await assert.rejects(pending, isInvalidResponse);
  assert.equal(requests[0].body.request_id, 'original-id');
  assert.equal(requests.length, 1);
});

test('adding an ID after submission leaves server-generated IDs valid', async (t) => {
  const {client, requests} = await endpoint(t, () => ({body: decision('dec-server-generated')}));
  const payload = request(undefined);
  const pending = client.decide(payload);
  payload.request_id = 'added-too-late';
  assert.equal((await pending).request_id, 'dec-server-generated');
  assert.equal(Object.hasOwn(requests[0].body, 'request_id'), false);
});

for (const swapped of [false, true]) {
  test(`concurrent requests ${swapped ? 'reject swapped' : 'preserve their own'} IDs`, async (t) => {
    let release;
    const bothArrived = new Promise((resolve) => { release = resolve; });
    const {client, requests} = await endpoint(t, async (body, count) => {
      if (count === 2) release();
      await bothArrived;
      const id = swapped ? (body.request_id === 'request-a' ? 'request-b' : 'request-a') : body.request_id;
      return {body: decision(id)};
    });
    const calls = ['request-a', 'request-b'].map((id) => client.decide(request(id)));
    if (swapped) {
      await Promise.all(calls.map((call) => assert.rejects(call, isInvalidResponse)));
    } else {
      assert.deepEqual((await Promise.all(calls)).map((result) => result.request_id), ['request-a', 'request-b']);
    }
    assert.equal(requests.length, 2);
    assert.deepEqual(requests.map(({body}) => body.request_id).sort(), ['request-a', 'request-b']);
  });
}

test('preserves structured HTTP errors for requests with an explicit ID', async (t) => {
  const {client, requests} = await endpoint(t, () => ({status: 409,
    body: {error: {code: 'duplicate_request', message: 'Already in flight'}}}));
  await assert.rejects(client.decide(request('my-request')),
    (error) => error instanceof JevAPIError && error.status === 409 && error.code === 'duplicate_request');
  assert.equal(requests.length, 1);
});
