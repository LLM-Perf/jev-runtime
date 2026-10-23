import assert from 'node:assert/strict';
import { once } from 'node:events';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { JevAPIError, JevClient } from '../dist/index.js';

const decision = JSON.parse(readFileSync(new URL('../../../tests/fixtures/decision-responses.json', import.meta.url))).valid.boolean;
const request = {model: 'fixture', input: {text: 'x'}};
const operations = {
  decide: {path: '/v1/decisions', method: 'POST', body: JSON.stringify(request), payload: decision,
    call: (client) => client.decide(request)},
  cancel: {path: '/v1/requests/owned%3Arequest/cancel', method: 'POST', body: '{}', payload: {cancelled: true},
    call: (client) => client.cancel('owned:request')},
  ready: {path: '/ready', method: 'GET', body: '', payload: {ready: true},
    call: (client) => client.ready()},
};

async function redirectServer(t, status, body, targetPayload) {
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    requests.push({path: request.url, method: request.method, body: Buffer.concat(chunks).toString()});
    if (request.url === '/redirect-target') {
      response.writeHead(200, {'Content-Type': 'application/json'});
      response.end(JSON.stringify(targetPayload));
    } else {
      response.writeHead(status, {Location: '/redirect-target', 'Content-Type': 'application/json'});
      response.end(body);
    }
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  return {requests, baseURL: `http://127.0.0.1:${server.address().port}`};
}

for (const [name, operation] of Object.entries(operations)) {
  // Native fetch changes POST to GET for 301/302/303 and replays the POST body
  // for 307/308. Every API must stop at the configured endpoint in both cases.
  for (const status of [301, 302, 303, 307, 308]) {
    test(`${name} rejects ${status} without following Location`, async (t) => {
      const error = {code: 'endpoint_moved', message: 'Configure the intended endpoint'};
      const {requests, baseURL} = await redirectServer(t, status, JSON.stringify({error}), operation.payload);
      const client = new JevClient({baseURL});
      await assert.rejects(operation.call(client),
        (e) => e instanceof JevAPIError && e.status === status && e.code === error.code && e.message === error.message);
      assert.deepEqual(requests, [{path: operation.path, method: operation.method, body: operation.body}]);
    });
  }

  for (const [kind, body] of [['success-shaped', JSON.stringify(operation.payload)], ['non-JSON', 'moved']]) {
    test(`${name} rejects a ${kind} redirect body as an HTTP error`, async (t) => {
      const {requests, baseURL} = await redirectServer(t, 303, body, operation.payload);
      const client = new JevClient({baseURL});
      await assert.rejects(operation.call(client),
        (e) => e instanceof JevAPIError && e.status === 303 && e.code === 'http_error');
      assert.deepEqual(requests, [{path: operation.path, method: operation.method, body: operation.body}]);
    });
  }
}

test('an opaque redirect remains an HTTP error when its status and body are hidden', async () => {
  let calls = 0;
  const client = new JevClient({baseURL: 'https://example.invalid', fetch: async (_, init) => {
    calls++;
    assert.equal(init.redirect, 'manual');
    // Browsers hide a manual redirect behind this filtered response. This is
    // a fetch double; native redirect behavior is exercised by the tests above.
    return {type: 'opaqueredirect', status: 0, ok: false,
      json: async () => { throw new SyntaxError('Unexpected end of JSON input'); }};
  }});
  await assert.rejects(client.ready(),
    (e) => e instanceof JevAPIError && e.status === 0 && e.code === 'http_error');
  assert.equal(calls, 1);
});
