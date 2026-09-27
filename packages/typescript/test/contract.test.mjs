import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { JevAPIError, JevClient, parseDecision } from '../dist/index.js';

const cases = JSON.parse(readFileSync(new URL('../../../tests/fixtures/decision-responses.json', import.meta.url)));

for (const [name, value] of Object.entries(cases.valid)) {
  test(`valid response: ${name}`, () => assert.deepEqual(parseDecision(value), value));
}
for (const {name, template, changes} of cases.invalid) {
  test(`reject corrupt response: ${name}`, () => {
    const payload = structuredClone(cases.valid[template]);
    for (const {path, value} of changes) {
      let target = payload;
      for (const part of path.slice(0, -1)) target = target[part];
      target[path.at(-1)] = value;
    }
    assert.throws(() => parseDecision(payload),
      (e) => e instanceof JevAPIError && e.status === 502 && e.code === 'invalid_response');
  });
}

test('cancellation preserves booleans and rejects malformed replies without retries', async () => {
  const replies = [Response.json({cancelled: false}), Response.json({cancelled: true}),
    Response.json({cancelled: 'false'}), new Response('not json'),
    Response.json({error: {code: 'generation_conflict', message: 'Changed'}}, {status: 409}),
    new Response('gateway unavailable', {status: 503}), Response.json({error: []}, {status: 502})];
  let calls = 0;
  const client = new JevClient({baseURL: 'http://localhost/plugins/jev-runtime', fetch: async (url) => {
    assert.equal(url, 'http://localhost/plugins/jev-runtime/v1/requests/owned%3Arequest/cancel');
    return replies[calls++];
  }});
  assert.equal(await client.cancel('owned:request'), false);
  assert.equal(await client.cancel('owned:request'), true);
  for (const [status, code] of [[502, 'invalid_response'], [502, 'invalid_response'],
    [409, 'generation_conflict'], [503, 'http_error'], [502, 'http_error']]) {
    await assert.rejects(client.cancel('owned:request'),
      (e) => e instanceof JevAPIError && e.status === status && e.code === code);
  }
  assert.equal(calls, replies.length);
});
